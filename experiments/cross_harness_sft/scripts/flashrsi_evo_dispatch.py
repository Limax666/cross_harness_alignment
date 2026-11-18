"""Route the one read-only harness tool without exposing host files."""
import re


class EvoRecorder:
    def __init__(self,native_recorder,snapshot):
        self.native=native_recorder
        self.snapshot=snapshot

    def tool(self,name,arguments):
        if name!='read_skill':
            return self.native.tool(name,arguments)
        skill=arguments.get('skill') if isinstance(arguments,dict) and set(arguments)=={'skill'} else None
        if not isinstance(skill,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',skill):
            return {'observation':'Invalid skill name.','invalid_call':True}
        entry=self.snapshot.files.get(f'skills/{skill}/SKILL.md')
        if entry is None:
            return {'observation':'Skill not found.','invalid_call':True}
        text=entry.content.decode('utf-8')
        match=re.match(r'^---\r?\n.*?\r?\n---(?:\r?\n|$)',text,re.DOTALL)
        return {'observation':text[match.end():].strip() if match else text,'invalid_call':False}
